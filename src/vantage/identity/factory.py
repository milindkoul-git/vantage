"""Resolving configuration into a working identity subsystem."""

from __future__ import annotations

from vantage.core.errors import ConfigError
from vantage.core.logging import get_logger
from vantage.identity.engine import IdentityEngine, IdentityParams
from vantage.identity.gallery import Gallery
from vantage.identity.recognizer import FaceRecognizer
from vantage.identity.store import IdentityStore

log = get_logger(__name__)

FACE_ADAPTERS = {"yunet": "face-detect", "sface": "face-embed"}
"""The face models this subsystem knows how to drive, and the task each fills.

These two do not go through :mod:`vantage.perception.adapters` - OpenCV's own
wrappers load them - so nothing else in the project would notice a catalog entry
naming a face adapter that does not exist, or a configuration pointing the face
detector at a YOLOX file. ``cv2.FaceDetectorYN.create`` on the wrong ONNX either
throws something opaque or, worse, loads and returns nonsense. This is the
registry that check reads from.
"""


def _require_face_model(key: str, role: str):
    """Resolve a catalog key and refuse it if it is not that kind of face model."""
    from vantage.perception.catalog import get_model_spec

    spec = get_model_spec(key)
    if spec.task != role:
        raise ConfigError(
            f"identity expects a {role!r} model but {key!r} is a {spec.task!r} model. "
            f"Available: {', '.join(k for k, v in FACE_ADAPTERS.items() if v == role)}"
        )
    return spec


def build_recognizer(config) -> FaceRecognizer:
    """Fetch and verify the two face models, then load them."""
    from vantage.perception.store import ModelStore

    store = ModelStore(config.model_dir)
    detector = store.ensure(
        _require_face_model(config.detector_model, "face-detect"),
        allow_download=config.allow_download,
    )
    embedder = store.ensure(
        _require_face_model(config.embedder_model, "face-embed"),
        allow_download=config.allow_download,
    )
    return FaceRecognizer(detector, embedder, score_threshold=config.face_score)


def build_identity_engine(config) -> tuple[IdentityEngine, IdentityStore]:
    """Construct the engine and the store that backs it.

    Returned as a pair because their lifetimes differ: the CLI opens the store
    alone to enrol, list and revoke, without loading either model.
    """
    store = IdentityStore(config.path)
    gallery = Gallery(store.load(), threshold=config.threshold, margin=config.margin)
    engine = IdentityEngine(
        build_recognizer(config),
        gallery,
        params=IdentityParams(
            interval=config.interval,
            min_votes=config.min_votes,
            max_attempts=config.max_attempts,
            reverify_interval=config.reverify_interval,
            min_face_fraction=config.min_face_fraction,
        ),
        store=store,
    )
    log.info(
        "identity ready",
        extra={
            "vantage_fields": {
                "enrolled": len(gallery),
                "names": ", ".join(gallery.names) or "nobody",
                "threshold": config.threshold,
                "store": str(store.path),
            }
        },
    )
    return engine, store
