"""The gallery: enrolled templates, and the decision of whether one matches.

Two thresholds, not one
-----------------------
A match must be **similar enough** and **clearly ahead of the runner-up**. Only
the first is the usual advice, and on its own it is what produces confident
misidentification: with two enrolled people who look somewhat alike, a face can
score 0.40 against both, clear the threshold, and be named after whichever
happened to score 0.401.

The margin requirement makes that outcome ``unknown`` instead, which is the
correct answer - the evidence genuinely does not distinguish them. Small
galleries make this *more* important rather than less, because with two
templates a coin flip is right half the time and looks like it works.

On the threshold value
----------------------
0.363 is the figure OpenCV Zoo publishes for SFace cosine matching, and it is
inherited rather than measured here: verifying it properly needs a labelled set
of faces this project has no business collecting. It is exposed in configuration
for that reason, and the honest statement is that it comes from the model's
authors and has not been re-derived on your camera.
"""

from __future__ import annotations

import numpy as np

from vantage.core.errors import ConfigError
from vantage.identity.contracts import UNKNOWN, Enrollment, IdentityMatch
from vantage.identity.recognizer import cosine

SFACE_COSINE_THRESHOLD = 0.363
"""The threshold published by OpenCV Zoo for these weights. Not measured here."""

DEFAULT_MARGIN = 0.05
"""How far ahead of the runner-up the winner must be.

Deliberately small. It is not trying to make matching harder; it is trying to
catch the specific case where two templates are nearly tied, which is when a
name is most likely to be the wrong one.
"""


class Gallery:
    """Enrolled templates and the matching rule."""

    def __init__(
        self,
        enrollments: list[Enrollment] | None = None,
        *,
        threshold: float = SFACE_COSINE_THRESHOLD,
        margin: float = DEFAULT_MARGIN,
    ) -> None:
        if not -1.0 <= threshold <= 1.0:
            raise ConfigError(f"identity.threshold must be in [-1, 1], got {threshold}")
        if margin < 0:
            raise ConfigError("identity.margin must be >= 0")
        self._threshold = threshold
        self._margin = margin
        self._entries: dict[str, Enrollment] = {}
        self._vectors: dict[str, np.ndarray] = {}
        for enrollment in enrollments or []:
            self.add(enrollment)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def margin(self) -> float:
        return self._margin

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def add(self, enrollment: Enrollment) -> None:
        self._entries[enrollment.name] = enrollment
        self._vectors[enrollment.name] = np.asarray(enrollment.template, dtype=np.float32)

    def remove(self, name: str) -> bool:
        existed = name in self._entries
        self._entries.pop(name, None)
        self._vectors.pop(name, None)
        return existed

    def get(self, name: str) -> Enrollment | None:
        return self._entries.get(name)

    def all(self) -> tuple[Enrollment, ...]:
        return tuple(self._entries[name] for name in self.names)

    def match(self, template: np.ndarray) -> IdentityMatch:
        """Compare one template against every enrolment.

        An empty gallery returns ``unknown`` rather than raising: a system with
        nobody enrolled should say it does not recognise anyone, which is true,
        rather than fail.
        """
        if not self._vectors:
            return IdentityMatch(name=UNKNOWN, similarity=0.0)

        scored = sorted(
            ((name, cosine(template, vector)) for name, vector in self._vectors.items()),
            key=lambda pair: -pair[1],
        )
        best_name, best_score = scored[0]
        runner_up, runner_up_score = scored[1] if len(scored) > 1 else (None, 0.0)

        if best_score < self._threshold:
            return IdentityMatch(
                name=UNKNOWN,
                similarity=best_score,
                runner_up=runner_up,
                runner_up_similarity=runner_up_score,
            )
        if runner_up is not None and (best_score - runner_up_score) < self._margin:
            # Similar enough to somebody, but not clearly to *this* somebody.
            # Naming the winner here is how a system confidently calls one
            # person by another's name.
            return IdentityMatch(
                name=UNKNOWN,
                similarity=best_score,
                runner_up=runner_up,
                runner_up_similarity=runner_up_score,
            )
        return IdentityMatch(
            name=best_name,
            similarity=best_score,
            runner_up=runner_up,
            runner_up_similarity=runner_up_score,
        )


def average_templates(templates: list[np.ndarray]) -> np.ndarray:
    """Combine several captures of one person into one template.

    Averaging then re-normalising, rather than keeping every sample and taking
    the best match. Several captures of one face from slightly different angles
    average toward what is stable about them; keeping them all and taking the
    maximum instead rewards whichever single capture happens to resemble the
    query, which is how a gallery starts matching strangers.
    """
    if not templates:
        raise ValueError("no templates to average")
    stacked = np.stack([np.asarray(t, dtype=np.float32).reshape(-1) for t in templates])
    mean = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 1e-9 else mean
