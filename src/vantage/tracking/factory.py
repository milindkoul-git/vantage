"""Building a tracker from configuration.

Separated from :mod:`vantage.tracking.bytetrack` so the tracker itself never
imports the configuration layer. That keeps the tracking package usable - and
more importantly, *testable and tunable* - with no config object anywhere in
sight, which is what lets the tuning harness construct thousands of parameter
sets directly.
"""

from __future__ import annotations

from vantage.config.schema import TrackingConfig
from vantage.tracking.base import Tracker
from vantage.tracking.bytetrack import ByteTracker, TrackerParams
from vantage.tracking.kalman import MotionNoise


def params_from_config(settings: TrackingConfig) -> TrackerParams:
    """Translate the config section into tracker parameters.

    Flat in the config, nested in the tracker: configuration files are more
    readable without a level of nesting for three noise values, while the
    tracker benefits from having them grouped into something it can pass around
    as one object.
    """
    return TrackerParams(
        high_threshold=settings.high_threshold,
        low_threshold=settings.low_threshold,
        init_threshold=settings.init_threshold,
        iou_high=settings.iou_high,
        iou_low=settings.iou_low,
        iou_tentative=settings.iou_tentative,
        min_hits=settings.min_hits,
        max_lost_s=settings.max_lost_s,
        max_step_s=settings.max_step_s,
        history=settings.history,
        class_aware=settings.class_aware,
        noise=MotionNoise(
            measurement=settings.measurement_noise,
            acceleration=settings.acceleration_noise,
            initial_velocity=settings.initial_velocity_noise,
            size_drift=settings.size_drift_noise,
        ),
    )


def build_tracker(settings: TrackingConfig) -> Tracker | None:
    """Construct the configured tracker, or ``None`` when tracking is disabled."""
    if not settings.enabled:
        return None
    return ByteTracker(params_from_config(settings))
