"""Resolving configuration into a running pose engine."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from vantage.core.clock import SYSTEM_CLOCK, Clock
from vantage.core.errors import ConfigError
from vantage.core.logging import get_logger
from vantage.perception.backends import create_backend
from vantage.perception.catalog import ModelSpec, get_model_spec
from vantage.perception.store import ModelStore
from vantage.pose.adapter import RTMPoseAdapter
from vantage.pose.engine import PoseEngine

log = get_logger(__name__)

_ADAPTERS = {"rtmpose": RTMPoseAdapter}
"""Pose adapters, kept separate from the detection registry because the two
solve different problems and share no interface. A single registry would have
to return objects that cannot be used interchangeably."""


def build_pose_engine(
    model: str = "rtmpose-s",
    *,
    backend: str = "auto",
    device: str = "auto",
    min_keypoint_confidence: float = 0.3,
    max_persons: int = 6,
    person_labels: Sequence[str] = ("person",),
    include_face_keypoints: bool = True,
    model_dir: str | Path = "models",
    threads: int = 0,
    allow_download: bool = True,
    clock: Clock = SYSTEM_CLOCK,
) -> PoseEngine:
    """Resolve a catalog key into a ready pose engine, fetching weights if needed."""
    spec: ModelSpec = get_model_spec(model)
    if spec.task != "pose":
        raise ConfigError(
            f"{spec.key!r} is a {spec.task} model, not a pose estimator. "
            "Pose models in the catalog: "
            f"{sorted(k for k, s in _pose_models().items())}"
        )
    try:
        adapter_cls = _ADAPTERS[spec.adapter]
    except KeyError:
        raise ConfigError(
            f"no pose adapter named {spec.adapter!r}; registered: {sorted(_ADAPTERS)}"
        ) from None

    store = ModelStore(model_dir)
    path = store.ensure(spec, allow_download=allow_download)

    adapter = adapter_cls(
        input_size=spec.input_size,
        labels=spec.labels,
        include_face_keypoints=include_face_keypoints,
    )
    inference_backend = create_backend(
        backend,
        path,
        device=device,
        threads=threads,
        input_shape=spec.input_size,
        input_shapes=adapter.static_input_shapes(),
    )
    engine = PoseEngine(
        adapter=adapter,
        backend=inference_backend,
        model_name=spec.key,
        license=spec.license,
        min_keypoint_confidence=min_keypoint_confidence,
        max_persons=max_persons,
        person_labels=person_labels,
        clock=clock,
    )
    log.info(
        "pose engine ready",
        extra={
            "vantage_fields": {
                "engine": engine.info.describe(),
                "face_keypoints": include_face_keypoints,
                "max_persons": max_persons,
            }
        },
    )
    return engine


def pose_engine_from_config(config, clock: Clock = SYSTEM_CLOCK) -> PoseEngine:
    """Build from a :class:`~vantage.config.schema.PoseConfig`."""
    return build_pose_engine(
        model=config.model,
        backend=config.backend,
        device=config.device,
        min_keypoint_confidence=config.min_keypoint_confidence,
        max_persons=config.max_persons,
        person_labels=config.classes,
        include_face_keypoints=config.include_face_keypoints,
        model_dir=config.model_dir,
        threads=config.threads,
        allow_download=config.allow_download,
        clock=clock,
    )


def _pose_models() -> dict[str, ModelSpec]:
    from vantage.perception.catalog import models_for_task

    return models_for_task("pose")
