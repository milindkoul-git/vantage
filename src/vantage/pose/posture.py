"""Posture from geometry: standing, sitting, crouching, lying.

Rules over a classifier, deliberately. A learned posture model would need
labelled data from this camera's viewpoint, would have to be retrained for the
next one, and would return a number nobody can audit. The four postures here
separate on two ratios that any human can check by hand against a drawn
skeleton, and when the ratios are not measurable the answer is
:attr:`~vantage.pose.contracts.Posture.UNKNOWN` rather than a guess.

The two measurements
--------------------
Both are vertical drops normalised by torso length, which makes them free of
scale and of distance from the camera:

``hip_knee``
    How far the knees fall below the hips. Near one when the thigh hangs
    vertically, near zero when it is horizontal.

``knee_ankle``
    How far the ankles fall below the knees - the same idea for the shank.

That gives a plainly separable plane::

    standing    hip_knee high, knee_ankle high     legs extended
    sitting     hip_knee low,  knee_ankle high     thigh horizontal, shank down
    crouching   hip_knee low,  knee_ankle low      both segments folded

Lying is settled before either ratio is computed, from the torso's angle to
vertical, because a horizontal body makes "below" meaningless - and then checked
against the detector's own box, which is an independent reading of the same
person. A horizontal torso inside a box taller than it is wide is a contradiction
rather than a fall; see :data:`LYING_MIN_BOX_ASPECT` for what that is worth,
measured.

What this cannot do
-------------------
It reads the *image*, not the world. A camera angled steeply down compresses
vertical drops and will eventually read a standing person as crouching; one
mounted sideways would read everyone as lying. There is no horizon estimate and
no calibration here to correct for it, so a tilted installation needs
:data:`LYING_ANGLE_DEG` reviewed rather than trusted. This is stated plainly
because the failure is quiet: the numbers stay confident while being wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vantage.pose.contracts import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    Pose,
    Posture,
)

LYING_ANGLE_DEG = 55.0
"""Torso tilt from vertical beyond which the body is called horizontal.

Generous on purpose. A person leaning over a desk reaches 30-40 degrees and must
not be reported as lying down, since in a later phase that is the difference
between a routine observation and a fall alert.
"""

LYING_MIN_BOX_ASPECT = 0.6
"""How wide a person's own box must be, relative to its height, before a
horizontal torso is accepted as lying down.

The skeleton and the box are independent readings of the same person, and where
they disagree the box is the more reliable of the two: it comes from the
detector, which had the whole frame, while the skeleton comes from a crop that
may be a dozen noisy pixels tall.

MEASURED, on 2,652 frames of five street clips in which nobody lies down and
nobody falls. The pose engine reported ``lying`` 77 times, every one of them
wrong. Their boxes ran 0.34-0.41 wide-over-tall: upright people, every one. So
this gate removes all 77 and touches no other posture.

The alternative was a confidence floor, and it was measured first and rejected:
the confidence distributions overlap almost entirely - false ``lying`` had a
median of 0.27 against standing's 0.20 - and a floor high enough to remove all
77 (0.40) also discarded 92.5% of every correct posture. A signal that cannot
separate the two is not a threshold problem.

0.6 rather than 1.0 leaves headroom for a real one: someone lying at an angle to
the camera, or with their legs out of frame, produces a box narrower than they
are long. The measured false readings sit far below even this.
"""

EXTENDED = 0.55
"""A limb segment dropping at least this fraction of a torso length is extended.

Between the standing cluster (around 0.9) and the folded ones (0.1-0.4), so
neither sits near the boundary.
"""

MIN_TORSO_FRACTION = 0.12
"""Torso length as a fraction of box height, below which nothing is decided.

A person square-on to the camera foreshortens; below this the torso is a few
pixels, every ratio divides by near zero, and the result is noise wearing a
confident label.
"""


@dataclass(frozen=True, slots=True)
class PostureEstimate:
    """A posture, how strongly it is believed, and why."""

    posture: Posture
    confidence: float
    reason: str

    @property
    def is_known(self) -> bool:
        return self.posture is not Posture.UNKNOWN


def classify(pose: Pose, min_keypoint_confidence: float = 0.3) -> PostureEstimate:
    """Classify one skeleton.

    Args:
        pose: The skeleton to read.
        min_keypoint_confidence: Score below which a landmark is treated as not
            observed. Landmarks are always emitted for all 17 joints, so without
            a floor here the rules would silently run on invented coordinates.
    """
    joint = _visible_joints(pose, min_keypoint_confidence)

    shoulder = _midpoint(joint, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip = _midpoint(joint, LEFT_HIP, RIGHT_HIP)
    if shoulder is None or hip is None:
        return PostureEstimate(
            Posture.UNKNOWN,
            0.0,
            _missing("torso", ("shoulder", shoulder), ("hip", hip)),
        )

    torso_length = math.dist(shoulder[:2], hip[:2])
    if torso_length < MIN_TORSO_FRACTION * max(pose.box.height, 1.0):
        return PostureEstimate(
            Posture.UNKNOWN,
            0.0,
            f"torso spans {torso_length:.0f}px of a {pose.box.height:.0f}px box, too "
            "foreshortened to measure",
        )

    torso_tilt = _tilt_from_vertical(shoulder[:2], hip[:2])
    torso_evidence = min(shoulder[2], hip[2])

    if torso_tilt >= LYING_ANGLE_DEG:
        box_aspect = pose.box.width / max(pose.box.height, 1e-6)
        if box_aspect < LYING_MIN_BOX_ASPECT:
            # The skeleton says horizontal and the box says upright. Reporting
            # the skeleton anyway is what produced false falls on far-field
            # footage: a person 40 pixels tall gives noisy joints, the torso
            # vector flips, and the one ALERT rule in the default set fires.
            # UNKNOWN rather than a guess at what they are instead - the
            # measurement is contradicted, not merely uncertain.
            return PostureEstimate(
                Posture.UNKNOWN,
                0.0,
                f"torso reads {torso_tilt:.0f} degrees from vertical, but the box is "
                f"{box_aspect:.2f} as wide as it is tall - too upright for a lying body",
            )
        return PostureEstimate(
            Posture.LYING,
            torso_evidence * _margin(torso_tilt, LYING_ANGLE_DEG, 25.0),
            f"torso {torso_tilt:.0f} degrees from vertical, box {box_aspect:.2f} wide",
        )

    knee = _midpoint(joint, LEFT_KNEE, RIGHT_KNEE)
    if knee is None:
        # The ordinary case for a desk webcam, and the honest answer: a seated
        # person and a standing one are identical from the hips up.
        return PostureEstimate(
            Posture.UNKNOWN,
            0.0,
            "upright torso, but no knees visible - standing and sitting are "
            "indistinguishable without them",
        )

    # Check individual leg drops to handle walking strides without mid-point dilution
    l_drop = (
        (joint[LEFT_KNEE][1] - joint[LEFT_HIP][1]) / torso_length
        if (LEFT_KNEE in joint and LEFT_HIP in joint)
        else None
    )
    r_drop = (
        (joint[RIGHT_KNEE][1] - joint[RIGHT_HIP][1]) / torso_length
        if (RIGHT_KNEE in joint and RIGHT_HIP in joint)
        else None
    )
    hip_knee_mid = (knee[1] - hip[1]) / torso_length
    hip_knee = max(
        [d for d in (l_drop, r_drop, hip_knee_mid) if d is not None], default=hip_knee_mid
    )
    ankle = _midpoint(joint, LEFT_ANKLE, RIGHT_ANKLE)

    if hip_knee >= EXTENDED:
        evidence = min(torso_evidence, knee[2])
        return PostureEstimate(
            Posture.STANDING,
            evidence * _margin(hip_knee, EXTENDED, 0.35),
            f"thigh drops {hip_knee:.2f} torso lengths, legs extended",
        )

    if ankle is None:
        # Thigh folded is already decisive against standing; sitting versus
        # crouching is not, so the shared parent is not reported as either.
        return PostureEstimate(
            Posture.UNKNOWN,
            0.0,
            f"thigh folded ({hip_knee:.2f}), but no ankles visible to separate "
            "sitting from crouching",
        )

    knee_ankle = (ankle[1] - knee[1]) / torso_length
    evidence = min(torso_evidence, knee[2], ankle[2])
    if knee_ankle >= EXTENDED:
        return PostureEstimate(
            Posture.SITTING,
            evidence * _margin(knee_ankle, EXTENDED, 0.35),
            f"thigh folded ({hip_knee:.2f}), shank extended ({knee_ankle:.2f})",
        )
    return PostureEstimate(
        Posture.CROUCHING,
        evidence * _margin(EXTENDED - knee_ankle, 0.0, 0.35),
        f"thigh and shank both folded ({hip_knee:.2f}, {knee_ankle:.2f})",
    )


def _visible_joints(pose: Pose, threshold: float) -> dict[int, tuple[float, float, float]]:
    """Landmarks above ``threshold``, keyed by full COCO index."""
    found: dict[int, tuple[float, float, float]] = {}
    for index in pose.visible(threshold):
        keypoint = pose.keypoint(index)
        if keypoint is not None:
            found[index] = (keypoint.x, keypoint.y, keypoint.confidence)
    return found


def _midpoint(
    joint: dict[int, tuple[float, float, float]], left: int, right: int
) -> tuple[float, float, float] | None:
    """Midpoint of a left/right pair, or the one side that was seen.

    Falling back to a single side is what keeps a person in profile classifiable
    at all - one shoulder and one hip occlude the other, and requiring both
    would return UNKNOWN for anyone not squarely facing the camera.
    """
    a, b = joint.get(left), joint.get(right)
    if a and b:
        return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, min(a[2], b[2]))
    return a or b


def _tilt_from_vertical(top: tuple[float, float], bottom: tuple[float, float]) -> float:
    """Angle in degrees between a segment and the image's vertical axis."""
    dx = bottom[0] - top[0]
    dy = bottom[1] - top[1]
    return math.degrees(math.atan2(abs(dx), abs(dy))) if (dx or dy) else 0.0


def _margin(value: float, boundary: float, span: float) -> float:
    """How far past a decision boundary a measurement sits, as 0 to 1.

    The reported confidence is this multiplied by the weakest landmark the rule
    depended on, so it degrades for both reasons a rule can be wrong: a joint
    that was barely seen, and a measurement that only just cleared the
    threshold. It is a heuristic score for ranking and display, not a calibrated
    probability, and nothing downstream should treat it as one.
    """
    return max(0.0, min(1.0, abs(value - boundary) / span))


def _missing(what: str, *parts: tuple[str, object]) -> str:
    absent = [name for name, value in parts if value is None]
    return f"no {what}: {' and '.join(absent)} not visible"
