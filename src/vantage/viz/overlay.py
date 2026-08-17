"""Detection overlay.

Pure rendering over a copy of the frame, with no dependency on how detections
were produced - the same code draws boxes from any future detector, and in
Phase 3 it will draw track ids alongside them.

Colour is derived deterministically from the class id, so a "person" is the
same colour in every frame and across runs. A palette that shuffled per frame
would make a video unreadable, and one keyed on detection order would flicker.
"""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

from vantage.perception.contracts import Detection, DetectionResult

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

    text = f"{detection.label} {detection.confidence:.2f}" if show_confidence else detection.label
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
