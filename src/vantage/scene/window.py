"""Temporal Observation Windows & Trajectory Encoding.

Provides sliding temporal buffers and deterministic feature extraction across:
1. Entity-level kinematics and skeletal dynamics (EntityTemporalWindow)
2. Scene-level multi-entity spatial distribution and convergence (SceneTemporalWindow)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from vantage.perception.contracts import BoundingBox
from vantage.pose.contracts import (
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    Pose,
    Posture,
)


@dataclass(frozen=True, slots=True)
class EntityObservationSample:
    """One instantaneous temporal observation for an entity."""

    timestamp: float
    box: BoundingBox
    normalized_box: tuple[float, float, float, float]  # (cx, cy, w, h) in [0, 1]
    foot_point: tuple[float, float]  # (fx, fy) in [0, 1]
    speed: float  # entity heights per second
    bearing_deg: float | None
    posture: Posture
    normalized_kpts: np.ndarray | None  # (17, 2) normalized to torso center and scale
    zones: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class KinematicFeatures:
    """Deterministic kinematic spatio-temporal features computed over an observation window."""

    duration_s: float
    sample_count: int
    mean_speed: float
    speed_variance: float
    max_acceleration: float
    directional_entropy: float  # 0.0 (straight) to 1.0 (highly erratic / omnidirectional)
    pacing_ratio: (
        float  # net_displacement / total_path_length (small = trapped/pacing in place)
    )
    path_length: float
    net_displacement: float


@dataclass(frozen=True, slots=True)
class SkeletalDynamics:
    """Deterministic skeletal dynamics computed over sequential pose frames."""

    hip_drop_rate: float  # vertical hip descent rate in heights/sec (positive = falling fast)
    mean_wrist_height: (
        float  # normalized wrist y relative to shoulder (< 0 = raised above shoulder)
    )
    wrist_velocity: float  # rate of wrist displacement
    is_prone: bool  # whether body aspect ratio and posture indicate lying down


class EntityTemporalWindow:
    """Sliding temporal observation buffer for a single tracked entity."""

    def __init__(self, max_samples: int = 60, max_span_s: float = 5.0) -> None:
        self.max_samples = max_samples
        self.max_span_s = max_span_s
        self.samples: deque[EntityObservationSample] = deque(maxlen=max_samples)

    def add(
        self,
        timestamp: float,
        box: BoundingBox,
        frame_width: int,
        frame_height: int,
        speed: float,
        bearing_deg: float | None,
        posture: Posture,
        pose: Pose | None = None,
        zones: tuple[str, ...] = (),
    ) -> None:
        """Add an observation frame and prune older samples outside temporal span."""
        w_f = max(1, frame_width)
        h_f = max(1, frame_height)

        # 1. Normalize bounding box and ground foot point
        cx = (box.x1 + box.x2) / (2.0 * w_f)
        cy = (box.y1 + box.y2) / (2.0 * h_f)
        bw = box.width / w_f
        bh = box.height / h_f
        fx = cx
        fy = box.y2 / h_f

        # 2. Skeletal landmarks
        kpt_array = None
        if pose is not None:
            kpt_array = np.zeros((17, 2), dtype=np.float32)
            for idx in range(17):
                kp = pose.keypoint(idx)
                if kp is not None and kp.confidence > 0.2:
                    kpt_array[idx, 0] = kp.x
                    kpt_array[idx, 1] = kp.y

        sample = EntityObservationSample(
            timestamp=timestamp,
            box=box,
            normalized_box=(cx, cy, bw, bh),
            foot_point=(fx, fy),
            speed=speed,
            bearing_deg=bearing_deg,
            posture=posture,
            normalized_kpts=kpt_array,
            zones=zones,
        )
        self.samples.append(sample)

        # Prune samples exceeding max_span_s
        while len(self.samples) > 2 and (
            timestamp - self.samples[0].timestamp > self.max_span_s
        ):
            self.samples.popleft()

    def extract_kinematics(self) -> KinematicFeatures:
        """Compute deterministic kinematic metrics over the current buffer."""
        n = len(self.samples)
        if n < 2:
            return KinematicFeatures(
                duration_s=0.0,
                sample_count=n,
                mean_speed=self.samples[0].speed if n == 1 else 0.0,
                speed_variance=0.0,
                max_acceleration=0.0,
                directional_entropy=0.0,
                pacing_ratio=1.0,
                path_length=0.0,
                net_displacement=0.0,
            )

        duration = max(0.001, self.samples[-1].timestamp - self.samples[0].timestamp)
        speeds = [s.speed for s in self.samples]
        mean_speed = float(np.mean(speeds))
        speed_var = float(np.var(speeds))

        # Accelerations
        accels: list[float] = []
        path_length = 0.0
        bearings: list[float] = []

        for i in range(1, n):
            raw_dt = self.samples[i].timestamp - self.samples[i - 1].timestamp
            if raw_dt <= 0.0:
                continue
            # If temporal gap exceeds 1.5s, treat as disconnected segment
            if raw_dt > 1.5:
                continue

            dt = max(
                0.02, raw_dt
            )  # Clamp lower bound to protect against jitter spikes at >50 FPS
            dv = abs(self.samples[i].speed - self.samples[i - 1].speed)
            accels.append(dv / dt)

            # Foot point displacement in normalized space or speed integration
            dx = self.samples[i].foot_point[0] - self.samples[i - 1].foot_point[0]
            dy = self.samples[i].foot_point[1] - self.samples[i - 1].foot_point[1]
            dist = math.hypot(dx, dy)
            if dist < 1e-5 and self.samples[i].speed > 0:
                dist = (self.samples[i].speed * dt) * 0.1
            path_length += dist

            bearing = self.samples[i].bearing_deg
            if bearing is not None:
                bearings.append(bearing)

        max_accel = max(accels) if accels else 0.0

        # Net displacement from start to end
        net_dx = self.samples[-1].foot_point[0] - self.samples[0].foot_point[0]
        net_dy = self.samples[-1].foot_point[1] - self.samples[0].foot_point[1]
        net_displacement = math.hypot(net_dx, net_dy)

        # Pacing ratio: requires genuine movement (path length > 0.01)
        pacing_ratio = (
            (net_displacement / max(path_length, 1e-5)) if path_length > 0.01 else 1.0
        )

        # Directional entropy over 8-bin bearing histogram (0 to 360 deg)
        directional_entropy = 0.0
        if len(bearings) >= 4 and path_length > 0.02:
            hist, _ = np.histogram(bearings, bins=8, range=(0, 360))
            probs = hist / np.sum(hist)
            # Shannon entropy normalized to [0, 1]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            directional_entropy = min(1.0, entropy / 3.0)  # log2(8) = 3.0

        return KinematicFeatures(
            duration_s=round(duration, 3),
            sample_count=n,
            mean_speed=round(mean_speed, 3),
            speed_variance=round(speed_var, 4),
            max_acceleration=round(max_accel, 3),
            directional_entropy=round(directional_entropy, 3),
            pacing_ratio=round(pacing_ratio, 3),
            path_length=round(path_length, 4),
            net_displacement=round(net_displacement, 4),
        )

    def extract_skeletal(self) -> SkeletalDynamics:
        """Compute skeletal dynamics over pose keypoints in the window."""
        valid_poses = [s for s in self.samples if s.normalized_kpts is not None]
        if len(valid_poses) < 2:
            last_posture = self.samples[-1].posture if self.samples else Posture.UNKNOWN
            return SkeletalDynamics(
                hip_drop_rate=0.0,
                mean_wrist_height=0.0,
                wrist_velocity=0.0,
                is_prone=last_posture is Posture.LYING,
            )

        first_sample = valid_poses[0]
        last_sample = valid_poses[-1]
        raw_dt = last_sample.timestamp - first_sample.timestamp
        if raw_dt > 2.5:  # Too large a gap between pose observations
            return SkeletalDynamics(
                hip_drop_rate=0.0,
                mean_wrist_height=0.0,
                wrist_velocity=0.0,
                is_prone=last_sample.posture in (Posture.LYING, "lying"),
            )
        dt = max(0.02, raw_dt)

        kpts_first = first_sample.normalized_kpts
        kpts_last = last_sample.normalized_kpts
        if kpts_first is None or kpts_last is None:
            # valid_poses already excluded these; narrowed rather than asserted
            # so the guarantee is checked instead of assumed.
            return SkeletalDynamics(
                hip_drop_rate=0.0,
                mean_wrist_height=0.0,
                wrist_velocity=0.0,
                is_prone=last_sample.posture in (Posture.LYING, "lying"),
            )
        initial_h = max(first_sample.box.height, 10.0)

        # Ensure hips were genuinely detected (non-zero coordinates) in both samples
        hip_drop_rate = 0.0
        if (kpts_first[LEFT_HIP, 1] > 0 or kpts_first[RIGHT_HIP, 1] > 0) and (
            kpts_last[LEFT_HIP, 1] > 0 or kpts_last[RIGHT_HIP, 1] > 0
        ):
            y_first = [y for y in (kpts_first[LEFT_HIP, 1], kpts_first[RIGHT_HIP, 1]) if y > 0]
            y_last = [y for y in (kpts_last[LEFT_HIP, 1], kpts_last[RIGHT_HIP, 1]) if y > 0]
            if y_first and y_last:
                hip_y_first = sum(y_first) / len(y_first)
                hip_y_last = sum(y_last) / len(y_last)
                hip_drop_rate = ((hip_y_last - hip_y_first) / initial_h) / dt

        # Wrist position relative to shoulder (only when both are present)
        mean_wrist_height = 0.0
        if (kpts_last[LEFT_WRIST, 1] > 0 or kpts_last[RIGHT_WRIST, 1] > 0) and (
            kpts_last[LEFT_SHOULDER, 1] > 0 or kpts_last[RIGHT_SHOULDER, 1] > 0
        ):
            w_y = [y for y in (kpts_last[LEFT_WRIST, 1], kpts_last[RIGHT_WRIST, 1]) if y > 0]
            s_y = [
                y for y in (kpts_last[LEFT_SHOULDER, 1], kpts_last[RIGHT_SHOULDER, 1]) if y > 0
            ]
            if w_y and s_y:
                mean_wrist_height = ((sum(w_y) / len(w_y)) - (sum(s_y) / len(s_y))) / initial_h

        # Wrist velocity
        wrist_velocity = 0.0
        if (kpts_first[LEFT_WRIST, 0] > 0 and kpts_last[LEFT_WRIST, 0] > 0) or (
            kpts_first[RIGHT_WRIST, 0] > 0 and kpts_last[RIGHT_WRIST, 0] > 0
        ):
            w_dx = kpts_last[LEFT_WRIST, 0] - kpts_first[LEFT_WRIST, 0]
            w_dy = kpts_last[LEFT_WRIST, 1] - kpts_first[LEFT_WRIST, 1]
            wrist_velocity = (math.hypot(w_dx, w_dy) / initial_h) / dt

        is_prone = last_sample.posture in (Posture.LYING, "lying") or (
            last_sample.box.width > 1.2 * last_sample.box.height
        )

        return SkeletalDynamics(
            hip_drop_rate=round(float(hip_drop_rate), 3),
            mean_wrist_height=round(float(mean_wrist_height), 3),
            wrist_velocity=round(float(wrist_velocity), 3),
            is_prone=bool(is_prone),
        )


@dataclass(frozen=True, slots=True)
class SceneObservationSample:
    """Collective spatial observation across all entities in a single frame."""

    timestamp: float
    camera_id: str
    entity_positions: tuple[tuple[str, float, float], ...]  # (entity_id, fx, fy)
    centroid: tuple[float, float]  # (cx, cy)
    spread_radius: float  # mean distance from centroid
    density: float  # entity count per unit area


@dataclass(frozen=True, slots=True)
class ConvergenceDynamics:
    """Collective multi-entity convergence or dispersion metrics over a sliding window."""

    entity_count: int
    centroid_velocity: float
    spread_rate: float  # positive = dispersing, negative = converging
    mean_pairwise_distance: float
    is_converging: bool
    is_dispersing: bool


class SceneTemporalWindow:
    """Sliding temporal buffer capturing collective multi-entity scene distributions."""

    def __init__(self, max_samples: int = 30, max_span_s: float = 3.0) -> None:
        self.max_samples = max_samples
        self.max_span_s = max_span_s
        self.samples: deque[SceneObservationSample] = deque(maxlen=max_samples)

    def add(
        self,
        timestamp: float,
        camera_id: str,
        entities: list[tuple[str, float, float]],  # (entity_id, fx, fy)
    ) -> None:
        """Record spatial distribution of all active entities in a frame."""
        if not entities:
            sample = SceneObservationSample(
                timestamp=timestamp,
                camera_id=camera_id,
                entity_positions=(),
                centroid=(0.5, 0.5),
                spread_radius=0.0,
                density=0.0,
            )
        else:
            cx = sum(e[1] for e in entities) / len(entities)
            cy = sum(e[2] for e in entities) / len(entities)
            spread = sum(math.hypot(e[1] - cx, e[2] - cy) for e in entities) / len(entities)
            density = len(entities) / max(0.01, math.pi * max(0.05, spread) ** 2)

            sample = SceneObservationSample(
                timestamp=timestamp,
                camera_id=camera_id,
                entity_positions=tuple(entities),
                centroid=(cx, cy),
                spread_radius=spread,
                density=density,
            )

        self.samples.append(sample)

        while len(self.samples) > 2 and (
            timestamp - self.samples[0].timestamp > self.max_span_s
        ):
            self.samples.popleft()

    def extract_convergence(self) -> ConvergenceDynamics:
        """Compute convergence or dispersion dynamics across the observation window."""
        n = len(self.samples)
        if n < 2 or not self.samples[-1].entity_positions:
            return ConvergenceDynamics(
                entity_count=len(self.samples[-1].entity_positions) if n > 0 else 0,
                centroid_velocity=0.0,
                spread_rate=0.0,
                mean_pairwise_distance=0.0,
                is_converging=False,
                is_dispersing=False,
            )

        first_sample = self.samples[0]
        last_sample = self.samples[-1]
        dt = max(0.01, last_sample.timestamp - first_sample.timestamp)

        # Centroid movement
        c_dx = last_sample.centroid[0] - first_sample.centroid[0]
        c_dy = last_sample.centroid[1] - first_sample.centroid[1]
        c_vel = math.hypot(c_dx, c_dy) / dt

        # Rate of spread change (negative = converging, positive = dispersing)
        spread_rate = (last_sample.spread_radius - first_sample.spread_radius) / dt

        # Pairwise distance in last sample
        pts = [pos[1:] for pos in last_sample.entity_positions]
        k = len(pts)
        mean_pdist = 0.0
        if k >= 2:
            dists = []
            for i in range(k):
                for j in range(i + 1, k):
                    dists.append(math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]))
            mean_pdist = sum(dists) / len(dists)

        is_converging = k >= 3 and spread_rate < -0.05 and last_sample.spread_radius < 0.25
        is_dispersing = k >= 3 and spread_rate > 0.10

        return ConvergenceDynamics(
            entity_count=k,
            centroid_velocity=round(c_vel, 3),
            spread_rate=round(spread_rate, 3),
            mean_pairwise_distance=round(mean_pdist, 3),
            is_converging=is_converging,
            is_dispersing=is_dispersing,
        )
