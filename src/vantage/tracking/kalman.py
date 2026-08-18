"""Constant-velocity Kalman filter over box geometry, with a real timestep.

State and parameterisation
--------------------------
Eight dimensions: ``[cx, cy, w, h, vx, vy, vw, vh]`` - centre, size, and the
rate of change of each. The reference SORT and ByteTrack implementations track
*aspect ratio* and height instead of width and height. Width is used here
because aspect ratio couples two independent measurements into one state
variable: a person who turns sideways changes width while their height is
constant, and the aspect-ratio form has to explain that as a correlated change
in both tracked dimensions. Tracking width directly lets the two vary
independently, which is what they actually do.

Why the timestep is an argument
-------------------------------
Almost every published tracker advances its filter by one unit per call and
calls that a frame. That assumption is false in this platform, twice over:

* ``detection.interval`` runs the detector on every Nth frame, so consecutive
  tracker steps can be 33 ms or 165 ms apart by configuration.
* Under ``latest`` backpressure the pipeline drops frames when the consumer
  falls behind, so the gap varies at runtime with system load.

A tracker that assumed uniform spacing would under-predict motion after a gap
by exactly the factor it was wrong about, and the object would be somewhere the
predicted box is not - which is an identity switch. :class:`~vantage.core.frame.Frame`
already carries ``capture_monotonic``, so the real elapsed time is available
and is used. The cost is that the process noise has to be derived rather than
tabulated, which is the next section.

Noise model
-----------
The process noise is the exact discretisation of a constant-velocity model
driven by white acceleration noise, per axis::

    Q_axis = [[dt^3/3, dt^2/2],
              [dt^2/2, dt   ]] * sigma_accel^2

This matters more than it looks. A hand-tuned per-frame noise table (what
DeepSORT uses) is only valid at the frame rate it was tuned at; the expression
above is correct at any ``dt``, which is precisely the property this platform
needs. Both noise magnitudes scale with the object's height so that the filter
behaves identically for a person near the camera and the same person far from
it - without that, one set of constants cannot fit both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vantage.perception.contracts import BoundingBox

NDIM = 4
"""Measured dimensions: cx, cy, w, h."""

VEL_DIM = 2
"""Velocity dimensions: the centre only. **Size is deliberately not given a
velocity**, which is the single most consequential decision in this file.

The reference SORT/ByteTrack filters do model size velocity, and doing the same
here produced a real, observed failure. As an object slides behind an occluder
the detector still sees it, but the visible part - and therefore the reported
box - shrinks rapidly. The filter reads that as a genuine, sustained shrinking
motion. The object then disappears completely, the filter extrapolates, and the
predicted box collapses toward zero. Measured on real footage: a coasting track
reached **1x1 pixels** after 40 frames. That is not merely an invisible box on
screen; a degenerate box has an IoU of essentially zero against anything, so
when the object reappeared it could never be re-associated and a new identity
was issued.

Physically, size velocity was never justified in the way centre velocity is. An
object's apparent size changes slowly, non-monotonically, and only because its
distance changed - whereas its position changes consistently and predictably.
Extrapolating size across a gap of no evidence asserts something nobody knows.
So size is filtered but never extrapolated: it is smoothed by measurements and
holds its last estimate while coasting."""

STATE_DIM = NDIM + VEL_DIM


@dataclass(frozen=True, slots=True)
class MotionNoise:
    """Tunable noise magnitudes, all relative to object height.

    Expressed as fractions of the tracked box height so a single set of values
    works across scales and resolutions. These are the parameters the tuning
    harness searches over; the defaults here are the searched result, not a
    guess (see :mod:`vantage.tracking.tuning`).
    """

    measurement: float = 0.05
    """Std of the detector's box error, as a fraction of height. 0.05 says a
    detector places a 200 px-tall person to within about 10 px, which matches
    what YOLOX-nano actually does on this hardware."""

    acceleration: float = 2.0
    """Std of unmodelled acceleration, in object heights per second squared.
    This is the single most important tuning knob: too low and the filter
    refuses to believe a genuine change of direction, too high and it chases
    detector jitter and the prediction is worthless during an occlusion."""

    initial_velocity: float = 1.0
    """Std of the velocity prior for a new track, in object heights per second.
    A new track has no velocity evidence at all, so this must be wide enough to
    cover realistic motion or the first few updates fight the prior."""

    size_drift: float = 0.2
    """Std of size change, in object heights per second, as a random walk.

    Size has no velocity (see :data:`VEL_DIM`), so this is how the filter stays
    willing to believe a genuine change of scale - somebody walking toward the
    camera - without ever extrapolating one. Too low and the box refuses to grow
    with an approaching object; too high and it tracks detector jitter."""

    def __post_init__(self) -> None:
        for name in ("measurement", "acceleration", "initial_velocity", "size_drift"):
            value = getattr(self, name)
            if not value > 0:
                raise ValueError(f"MotionNoise.{name} must be positive, got {value}")


class KalmanBoxFilter:
    """Tracks one box. Owns mutable state; never shared between tracks."""

    __slots__ = ("_mean", "_covariance", "_noise")

    def __init__(self, box: BoundingBox, noise: MotionNoise | None = None) -> None:
        self._noise = noise or MotionNoise()
        cx, cy = box.center
        height = max(box.height, 1.0)

        self._mean = np.zeros(STATE_DIM, dtype=np.float64)
        self._mean[:NDIM] = (cx, cy, max(box.width, 1.0), height)

        # Position and size are known about as well as the detector can report
        # them; velocity is not known at all yet, hence the much wider prior.
        position_std = self._noise.measurement * height
        velocity_std = self._noise.initial_velocity * height
        self._covariance = np.diag(
            np.square(
                np.array(
                    [position_std] * NDIM + [velocity_std] * VEL_DIM,
                    dtype=np.float64,
                )
            )
        )

    @property
    def mean(self) -> np.ndarray:
        """Current state estimate; a copy, so callers cannot corrupt the filter."""
        return self._mean.copy()

    @property
    def box(self) -> BoundingBox:
        """Current estimate as a bounding box.

        Width and height are floored at one pixel: the filter is unconstrained
        and a fast-shrinking box can cross zero, which would produce an inverted
        box and fail :class:`BoundingBox`'s own validation far from the cause.
        """
        cx, cy, w, h = self._mean[:NDIM]
        half_w = max(w, 1.0) / 2.0
        half_h = max(h, 1.0) / 2.0
        return BoundingBox(x1=cx - half_w, y1=cy - half_h, x2=cx + half_w, y2=cy + half_h)

    @property
    def velocity(self) -> tuple[float, float]:
        """Centre velocity in pixels per second."""
        return float(self._mean[NDIM]), float(self._mean[NDIM + 1])

    def predict(self, dt: float) -> None:
        """Advance the state by ``dt`` seconds.

        A non-positive ``dt`` is a no-op rather than an error: a monotonic clock
        can legitimately report two events in the same tick, and refusing to
        advance is the correct response to "no time has passed". Advancing by a
        negative amount would run the model backwards.
        """
        if dt <= 0.0:
            return

        # Only the centre is advanced by its velocity. Width and height carry
        # forward unchanged - see VEL_DIM for why extrapolating them is wrong.
        transition = np.eye(STATE_DIM, dtype=np.float64)
        transition[0, NDIM] = dt
        transition[1, NDIM + 1] = dt

        self._mean = transition @ self._mean
        self._covariance = transition @ self._covariance @ transition.T + self._process_noise(dt)

    def update(self, box: BoundingBox) -> None:
        """Correct the state with a measured box."""
        measurement = np.array(
            [*box.center, max(box.width, 1.0), max(box.height, 1.0)], dtype=np.float64
        )

        observation = np.zeros((NDIM, STATE_DIM), dtype=np.float64)
        observation[:, :NDIM] = np.eye(NDIM)

        innovation_cov = (
            observation @ self._covariance @ observation.T + self._measurement_noise()
        )
        # solve() rather than inv(): a 4x4 inverse is cheap either way, but the
        # solve is better conditioned, and this runs on every matched track on
        # every frame.
        gain = np.linalg.solve(innovation_cov.T, (self._covariance @ observation.T).T).T

        self._mean = self._mean + gain @ (measurement - observation @ self._mean)

        # Joseph form. The textbook (I - KH)P is algebraically identical but
        # loses symmetry to floating-point error over thousands of updates, and
        # an asymmetric covariance eventually goes non-positive-definite and
        # produces silently wrong gains. A long-lived track is exactly the case
        # that reaches thousands of updates.
        identity = np.eye(STATE_DIM, dtype=np.float64)
        factor = identity - gain @ observation
        self._covariance = (
            factor @ self._covariance @ factor.T + gain @ self._measurement_noise() @ gain.T
        )

    def _height(self) -> float:
        return max(float(self._mean[3]), 1.0)

    def _process_noise(self, dt: float) -> np.ndarray:
        """Exact discretised constant-velocity noise for a step of ``dt``."""
        sigma = self._noise.acceleration * self._height()
        variance = sigma * sigma

        pos_pos = (dt**3) / 3.0 * variance
        pos_vel = (dt**2) / 2.0 * variance
        vel_vel = dt * variance

        noise = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
        for axis in range(VEL_DIM):  # cx and cy, each with its velocity
            noise[axis, axis] = pos_pos
            noise[axis, NDIM + axis] = pos_vel
            noise[NDIM + axis, axis] = pos_vel
            noise[NDIM + axis, NDIM + axis] = vel_vel

        # Width and height are a random walk rather than a velocity model, so
        # their uncertainty grows linearly with time instead of cubically. That
        # is what lets the box stay believable across a long occlusion instead
        # of drifting to a degenerate size.
        size_variance = (self._noise.size_drift * self._height()) ** 2 * dt
        for axis in range(VEL_DIM, NDIM):
            noise[axis, axis] = size_variance
        return noise

    def _measurement_noise(self) -> np.ndarray:
        std = self._noise.measurement * self._height()
        return np.eye(NDIM, dtype=np.float64) * (std * std)
