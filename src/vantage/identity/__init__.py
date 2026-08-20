"""Optional identity resolution.

The one subsystem in this platform that answers *who*, and the one the
specification fenced most carefully. Those fences are implemented rather than
documented:

* **Tracking never depends on it.** The pipeline runs identically with identity
  off; a name attaches to an existing anonymous ``entity_id`` or does not.
* **Enrolment is an act, not an observation.** No code path leads from the live
  pipeline to the gallery. A face becomes a name because a person at the machine
  ran a command with an explicit consent flag.
* **Templates, never images.** A 128-dimensional vector is stored. That is
  biometric data - it is not anonymised - but the database cannot be turned back
  into photographs.
* **Unknown is a real answer.** Below the threshold, or too close to a second
  candidate, the answer is unknown rather than the nearest guess.
* **Everything is audited**, including rejections, so "did this system decide it
  knew me" has an answer.
* **Its own database**, separate from the observation store, so biometrics can be
  handled and deleted on their own terms.

Models are YuNet (MIT) and SFace (Apache-2.0) from OpenCV Zoo. ArcFace via
InsightFace is the usual choice and its weights are "for non-commercial research
purposes only", which fails the same licence gate that ruled out YOLO-World and
Ultralytics pose.
"""

from vantage.identity.contracts import (
    EMBEDDING_DIM,
    UNKNOWN,
    AuditAction,
    AuditRecord,
    Enrollment,
    EntityIdentity,
    IdentityMatch,
    IdentityResult,
)

__all__ = [
    "EMBEDDING_DIM",
    "UNKNOWN",
    "AuditAction",
    "AuditRecord",
    "Enrollment",
    "EntityIdentity",
    "FaceRecognizer",
    "Gallery",
    "IdentityEngine",
    "IdentityMatch",
    "IdentityParams",
    "IdentityResult",
    "IdentityStore",
    "build_identity_engine",
]


def __getattr__(name: str):
    if name in ("IdentityEngine", "IdentityParams"):
        from vantage.identity import engine

        return getattr(engine, name)
    if name == "Gallery":
        from vantage.identity.gallery import Gallery

        return Gallery
    if name == "FaceRecognizer":
        from vantage.identity.recognizer import FaceRecognizer

        return FaceRecognizer
    if name == "IdentityStore":
        from vantage.identity.store import IdentityStore

        return IdentityStore
    if name == "build_identity_engine":
        from vantage.identity.factory import build_identity_engine

        return build_identity_engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
