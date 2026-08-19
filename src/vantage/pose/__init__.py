"""Human pose estimation.

Consumes tracks and produces skeletons bound to the tracker's anonymous entity
ids. Nothing here identifies anyone: see :mod:`vantage.pose.contracts` for what
the head landmarks are and are not.

Imports are deferred so that ``import vantage.pose`` costs nothing until a pose
engine is actually built - the adapter pulls in OpenCV and the factory pulls in
an inference runtime, neither of which an ingestion-only install has.
"""

from vantage.pose.contracts import (
    FACE_KEYPOINTS,
    KEYPOINT_NAMES,
    SKELETON,
    Keypoint,
    Pose,
    PoseResult,
    Posture,
    empty_pose_result,
)

__all__ = [
    "FACE_KEYPOINTS",
    "KEYPOINT_NAMES",
    "Keypoint",
    "Pose",
    "PoseResult",
    "PostureEstimate",
    "Posture",
    "SKELETON",
    "build_pose_engine",
    "classify",
    "empty_pose_result",
    "PoseEngine",
]


def __getattr__(name: str):
    if name in ("classify", "PostureEstimate"):
        from vantage.pose import posture

        return getattr(posture, name)
    if name == "PoseEngine":
        from vantage.pose.engine import PoseEngine

        return PoseEngine
    if name == "build_pose_engine":
        from vantage.pose.factory import build_pose_engine

        return build_pose_engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
