"""Detection overlay.

Pure rendering over a copy of the frame, with no dependency on how detections
or tracks were produced - the same code draws boxes from any detector and any
tracker behind the shared contracts.

Colour is always derived deterministically from a stable id, never from
position in a list. A palette keyed on detection order would flicker every
frame; one that shuffled per run would make two recordings incomparable. Which
id is used differs by overlay and is explained at each function.
"""

from __future__ import annotations

import colorsys
from typing import TYPE_CHECKING

import cv2
import numpy as np

from vantage.perception.contracts import Detection, DetectionResult

if TYPE_CHECKING:  # tracking is optional at render time; the overlay must not
    # make it an import-time requirement for plain detection display.
    from vantage.activity.contracts import ActivityResult
    from vantage.pose.contracts import PoseResult
    from vantage.spatial.contracts import SpatialResult, Zone
    from vantage.tracking.contracts import Track, TrackingResult

from vantage.activity.contracts import Activity
from vantage.pose.contracts import SKELETON, Posture
from vantage.spatial.contracts import Relation

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def class_color(class_id: int) -> tuple[int, int, int]:
    """A stable, well-separated BGR colour for a class index.

    The golden-ratio hue step keeps consecutive class ids visually distinct
    instead of shading into each other the way a linear ramp would.
    """
    hue = (class_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_detections(
    image: np.ndarray,
    result: DetectionResult,
    *,
    show_confidence: bool = True,
    thickness: int = 2,
    stale: bool = False,
) -> np.ndarray:
    """Draw ``result``'s boxes and return the annotated image.

    Ownership, stated precisely because it is easy to get wrong: a **writeable**
    ``image`` is drawn on **in place** and returned, while a read-only one is
    copied first. The run loop always passes a buffer it already owns (from the
    HUD or ``Frame.editable_copy``), so drawing in place avoids a second
    full-frame copy per frame - 6 MB at 1080p. Pass ``image.copy()`` if you need
    the original preserved.

    Args:
        stale: Mark boxes as carried over from an earlier frame. When detection
            runs every Nth frame, showing last pass's boxes as if they were
            current would misrepresent them, so they are drawn dashed instead.
    """
    canvas = image if image.flags.writeable else image.copy()
    scale = max(0.4, min(0.7, canvas.shape[1] / 1600.0))

    for detection in result:
        _draw_one(canvas, detection, scale, thickness, show_confidence, stale)
    return canvas


def _draw_one(
    canvas: np.ndarray,
    detection: Detection,
    scale: float,
    thickness: int,
    show_confidence: bool,
    stale: bool,
) -> None:
    x1, y1, x2, y2 = detection.box.to_int()
    color = class_color(detection.class_id)

    if stale:
        _dashed_rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
    else:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    text = (
        f"{detection.label} {detection.confidence:.2f}" if show_confidence else detection.label
    )
    (text_w, text_h), baseline = cv2.getTextSize(text, _FONT, scale, 1)

    # Put the label inside the box when there is no room above it, so labels on
    # objects at the top edge stay visible instead of being clipped away.
    label_top = y1 - text_h - baseline - 2
    if label_top < 0:
        label_top = y1 + 2
    label_bottom = label_top + text_h + baseline + 2

    cv2.rectangle(canvas, (x1, label_top), (x1 + text_w + 6, label_bottom), color, -1)
    cv2.putText(
        canvas,
        text,
        (x1 + 3, label_bottom - baseline),
        _FONT,
        scale,
        _readable_text_color(color),
        1,
        cv2.LINE_AA,
    )


def draw_tracks(
    image: np.ndarray,
    result: TrackingResult,
    *,
    show_confidence: bool = False,
    thickness: int = 2,
    stale: bool = False,
    trail: bool = True,
) -> np.ndarray:
    """Draw tracked objects with their anonymous entity ids and motion trails.

    Same ownership contract as :func:`draw_detections`: a writeable ``image`` is
    drawn on in place and returned.

    Colour is keyed on ``track_id`` rather than on class, which is the opposite
    of the detection overlay and deliberately so. Once objects have identity,
    the question a viewer is asking changes from "what are these things" to
    "which one is which", and colouring three people identically because they
    share a class makes exactly the thing you are watching for - a swap -
    invisible.

    Args:
        trail: Draw each track's recent path. This is the quickest way to see an
            identity switch by eye: the trail visibly teleports between objects.
    """
    canvas = image if image.flags.writeable else image.copy()
    scale = max(0.4, min(0.7, canvas.shape[1] / 1600.0))

    for track in result:
        color = track_color(track.track_id)
        # A coasting track is a prediction, not an observation, and is drawn
        # dashed so the display never implies evidence that does not exist.
        coasting = stale or track.is_coasting
        if trail and len(track.history) > 1:
            _draw_trail(canvas, track.history, color)
        _draw_track_box(canvas, track, color, scale, thickness, show_confidence, coasting)
    return canvas


def track_color(track_id: int) -> tuple[int, int, int]:
    """A stable, well-separated BGR colour for a track id."""
    hue = ((track_id * 0.61803398875) + 0.35) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _draw_track_box(
    canvas: np.ndarray,
    track: Track,
    color: tuple[int, int, int],
    scale: float,
    thickness: int,
    show_confidence: bool,
    coasting: bool,
) -> None:
    x1, y1, x2, y2 = track.box.to_int()
    if coasting:
        _dashed_rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
    else:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    text = track.entity_id
    if show_confidence:
        text += f" {track.confidence:.2f}"
    if coasting:
        text += " ?"

    (text_w, text_h), baseline = cv2.getTextSize(text, _FONT, scale, 1)
    label_top = y1 - text_h - baseline - 2
    if label_top < 0:
        label_top = y1 + 2
    label_bottom = label_top + text_h + baseline + 2

    cv2.rectangle(canvas, (x1, label_top), (x1 + text_w + 6, label_bottom), color, -1)
    cv2.putText(
        canvas,
        text,
        (x1 + 3, label_bottom - baseline),
        _FONT,
        scale,
        _readable_text_color(color),
        1,
        cv2.LINE_AA,
    )


def _draw_trail(
    canvas: np.ndarray, history: tuple[tuple[float, float], ...], color: tuple[int, int, int]
) -> None:
    """Draw the path a track has taken, fading toward the oldest point.

    The fade is what makes it readable as a direction rather than a smear.
    """
    points = [(int(round(x)), int(round(y))) for x, y in history]
    total = len(points)
    for index in range(1, total):
        weight = index / total
        faded = tuple(int(channel * (0.25 + 0.75 * weight)) for channel in color)
        cv2.line(canvas, points[index - 1], points[index], faded, 1, cv2.LINE_AA)


def _dashed_rectangle(
    canvas: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash: int = 8,
) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    for x in range(x1, x2, dash * 2):
        cv2.line(canvas, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(canvas, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(canvas, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(canvas, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def _readable_text_color(background: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black or white, whichever stays legible on ``background``.

    Class colours span the full hue circle, so a fixed text colour would vanish
    against roughly half of them.
    """
    blue, green, red = background
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def draw_poses(
    image: np.ndarray,
    result: PoseResult,
    *,
    min_confidence: float = 0.3,
    thickness: int = 2,
    show_posture: bool = True,
) -> np.ndarray:
    """Draw skeletons over the tracks they belong to.

    Same ownership contract as :func:`draw_detections`: a writeable ``image`` is
    drawn on in place and returned.

    Colour is keyed on ``track_id``, matching :func:`draw_tracks`, so a skeleton
    and its box are visibly the same entity.

    Joints below ``min_confidence`` are **not drawn at all**, and neither is any
    bone that depends on one. This is the same threshold the posture classifier
    treats as "not observed", so what you see is what the rules saw. Drawing
    low-confidence joints faintly was tried first and was worse than useless: an
    invented ankle at the bottom of the crop looks exactly like a real one that
    is merely dim, and the picture stopped agreeing with the classification
    printed next to it.
    """
    canvas = image if image.flags.writeable else image.copy()
    scale = max(0.4, min(0.7, canvas.shape[1] / 1600.0))

    for pose in result:
        color = track_color(pose.track_id)
        points: dict[int, tuple[int, int]] = {}
        for index in pose.visible(min_confidence):
            keypoint = pose.keypoint(index)
            if keypoint is not None:
                points[index] = (int(round(keypoint.x)), int(round(keypoint.y)))

        for start, end in SKELETON:
            if start in points and end in points:
                cv2.line(canvas, points[start], points[end], color, thickness, cv2.LINE_AA)
        for point in points.values():
            cv2.circle(canvas, point, thickness + 1, color, -1, cv2.LINE_AA)

        if show_posture and pose.posture is not Posture.UNKNOWN:
            label = f"{pose.posture.value} {pose.posture_confidence:.2f}"
            x1, y1, _, _ = pose.box.to_int()
            _draw_label(canvas, label, (x1, max(0, y1 - 4)), color, scale)
    return canvas


def _draw_label(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
) -> None:
    """A filled caption with contrast-picked text, anchored above ``origin``."""
    (width, height), _ = cv2.getTextSize(text, _FONT, scale, 1)
    x, y = origin
    top = max(0, y - height - 6)
    cv2.rectangle(canvas, (x, top), (x + width + 6, top + height + 6), color, -1)
    cv2.putText(
        canvas,
        text,
        (x + 3, top + height + 1),
        _FONT,
        scale,
        _readable_text_color(color),
        1,
        cv2.LINE_AA,
    )


def draw_activities(
    image: np.ndarray,
    result: ActivityResult,
    tracking: TrackingResult,
    *,
    show_idle: bool = False,
) -> np.ndarray:
    """Label each entity with what it is doing, under its box.

    Same ownership contract as :func:`draw_detections`.

    Placed **below** the box, because the entity id and class already sit above
    it and stacking three captions on one edge makes all of them unreadable in a
    crowd.

    ``idle`` is hidden by default. It is a real observation and worth carrying in
    the record, but printing "idle" over everyone standing still turns the
    display into noise and buries the one entity that is doing something.
    """
    canvas = image if image.flags.writeable else image.copy()
    scale = max(0.4, min(0.7, canvas.shape[1] / 1600.0))
    boxes = {track.track_id: track.box for track in tracking.tracks}

    for entity in result:
        box = boxes.get(entity.track_id)
        if box is None:
            continue
        primary = entity.primary
        if primary is None:
            continue
        if primary.activity is Activity.IDLE and not show_idle:
            continue

        _, _, _, y2 = box.to_int()
        x1 = box.to_int()[0]
        label = f"{primary.activity.value} {primary.confidence:.2f}"
        # A transient event gets the alert colour rather than the track's, so a
        # fall does not blend into whatever hue that person happens to own.
        color = (0, 165, 255) if primary.activity.is_transient else track_color(entity.track_id)
        _draw_label(canvas, label, (x1, min(canvas.shape[0] - 2, y2 + 18)), color, scale)
    return canvas


def draw_zones(
    image: np.ndarray,
    zones: tuple[Zone, ...],
    result: SpatialResult | None = None,
    *,
    thickness: int = 2,
) -> np.ndarray:
    """Draw zone polygons and their current occupancy.

    Same ownership contract as :func:`draw_detections`.

    Outlines rather than filled regions. A translucent fill was tried first and
    was worse: it tints everything inside, including the people the operator is
    watching, and two overlapping zones compound into a colour that belongs to
    neither. An outline says exactly where the boundary is and obscures nothing.
    """
    canvas = image if image.flags.writeable else image.copy()
    height, width = canvas.shape[:2]
    scale = max(0.4, min(0.7, width / 1600.0))
    occupancy = result.occupancy() if result is not None else {}

    for index, zone in enumerate(zones):
        color = class_color(index * 7 + 3)
        points = np.array(
            [[int(x * width), int(y * height)] for x, y in zone.points], dtype=np.int32
        )
        cv2.polylines(canvas, [points], True, color, thickness, cv2.LINE_AA)

        count = occupancy.get(zone.name, 0)
        label = zone.name if not count else f"{zone.name} ({count})"
        anchor = points[points[:, 1].argmin()]
        _draw_label(canvas, label, (int(anchor[0]), int(anchor[1])), color, scale)
    return canvas


def draw_relations(
    image: np.ndarray,
    result: SpatialResult,
    tracking: TrackingResult,
    *,
    thickness: int = 2,
) -> np.ndarray:
    """Draw a line between related entities, labelled with the relation.

    Only the relations worth a line are drawn: `near` is left out, because in a
    group everyone is near everyone and the frame turns into a mesh that hides
    the one relation that matters. Interaction and approach are drawn, since
    those are the claims a viewer would want to check by eye.
    """
    canvas = image if image.flags.writeable else image.copy()
    scale = max(0.4, min(0.7, canvas.shape[1] / 1600.0))
    boxes = {track.track_id: track.box for track in tracking.tracks}

    for relation in result.relations:
        if relation.relation is Relation.NEAR:
            continue
        first = boxes.get(relation.subject_track)
        second = boxes.get(relation.object_track)
        if first is None or second is None:
            continue

        start = tuple(int(v) for v in first.bottom_center)
        end = tuple(int(v) for v in second.bottom_center)
        # Amber for interaction, matching the transient-event colour the
        # activity overlay uses: both are claims rather than measurements.
        color = (0, 165, 255) if relation.relation is Relation.INTERACTING else (200, 200, 200)
        cv2.line(canvas, start, end, color, thickness, cv2.LINE_AA)
        midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        _draw_label(
            canvas,
            f"{relation.relation.value} {relation.confidence:.2f}",
            midpoint,
            color,
            scale,
        )
    return canvas
