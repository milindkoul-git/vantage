"""Enrolment: the only way a face becomes a name.

There is deliberately no path from the running pipeline to this module. Nothing
the camera sees can add an identity. Enrolment happens because a person at the
machine ran a command, named someone, and affirmed that the someone agreed to
it - which is what makes this an identity system rather than a surveillance one.

The consent flag is not decoration
----------------------------------
It is refused without ``consent=True``, and the CLI's help text says what the
flag asserts rather than describing it as a switch. A required argument that
states its meaning is the difference between a person choosing to enrol someone
and a person discovering afterwards that they had.

What is stored
--------------
A 128-dimensional vector, averaged over several captures, and nothing else. No
crop is written to disk at any point - not to a cache, not to a temporary file.
That is not the same as anonymity: a face template is biometric data and the
README says so plainly. It does mean that the database cannot be turned back
into photographs of anyone.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from vantage.core.errors import ConfigError, VantageError
from vantage.core.logging import get_logger
from vantage.identity.contracts import Enrollment
from vantage.identity.gallery import average_templates
from vantage.identity.recognizer import FaceRecognizer

log = get_logger(__name__)

CONSENT_REQUIRED = (
    "enrolment requires --consent. It asserts that the person being enrolled "
    "knows about it and agreed to it. This system will not add a face to its "
    "gallery on anyone's behalf."
)


def enroll_from_images(
    recognizer: FaceRecognizer,
    name: str,
    paths: list[str | Path],
    *,
    consent: bool,
    note: str = "",
) -> Enrollment:
    """Build one enrolment from image files."""
    _require_consent(consent)
    _require_name(name)

    templates: list[np.ndarray] = []
    skipped: list[str] = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise VantageError(f"could not read {path}")
        template = recognizer.template_for(image)
        if template is None:
            skipped.append(str(path))
            continue
        templates.append(template)

    if not templates:
        raise VantageError(
            f"no usable face found in {len(paths)} image(s) for {name!r}. "
            "The face must be large enough and clear enough to align; "
            f"skipped: {', '.join(skipped) or 'none'}"
        )
    if skipped:
        log.warning(
            "some images had no usable face",
            extra={"vantage_fields": {"name": name, "skipped": ", ".join(skipped)}},
        )
    return _build(name, templates, note or f"{len(templates)} image(s)")


def enroll_from_camera(
    recognizer: FaceRecognizer,
    name: str,
    source_uri: str,
    *,
    consent: bool,
    samples: int = 8,
    max_frames: int = 400,
    note: str = "",
    progress: Callable[[int, int], None] | None = None,
) -> Enrollment:
    """Capture several faces from a live source and enrol them.

    Takes captures spread across frames rather than the first N in a row: eight
    consecutive frames of a still person are eight copies of one pose, which
    averages to exactly that pose and generalises no better than a single
    capture would.
    """
    _require_consent(consent)
    _require_name(name)
    if samples < 1:
        raise ConfigError("samples must be >= 1")

    from vantage.config.schema import SourceConfig
    from vantage.ingestion.registry import create_source

    source = create_source(SourceConfig(uri=source_uri))
    templates: list[np.ndarray] = []
    spacing = max(1, max_frames // (samples * 4))

    with source as opened:
        for index in range(max_frames):
            try:
                frame = opened.read()
            except Exception as exc:
                log.debug("frame read failed", extra={"vantage_fields": {"error": str(exc)}})
                continue
            if index % spacing:
                continue
            template = recognizer.template_for(frame.image)
            if template is None:
                continue
            templates.append(template)
            if progress:
                progress(len(templates), samples)
            if len(templates) >= samples:
                break

    if not templates:
        raise VantageError(
            f"no face was captured for {name!r} in {max_frames} frames. "
            "Check the camera sees a face, well lit and facing forward."
        )
    return _build(name, templates, note or f"{len(templates)} camera captures")


def _build(name: str, templates: list[np.ndarray], note: str) -> Enrollment:
    return Enrollment(
        name=name.strip(),
        template=tuple(float(v) for v in average_templates(templates)),
        samples=len(templates),
        enrolled_at=time.time(),
        note=note,
    )


def _require_consent(consent: bool) -> None:
    if not consent:
        raise ConfigError(CONSENT_REQUIRED)


def _require_name(name: str) -> None:
    if not name or not name.strip():
        raise ConfigError("enrolment needs a name")
    if len(name.strip()) > 64:
        raise ConfigError("an enrolment name must be 64 characters or fewer")
